# Version 2: implementation plan

**Status: stage 1 APPROVED 2026-10-19.** The owner decided D1–D7 the same day, every one as recommended (§1.2), and
approved the plan. **Built, reviewed and regenerated 2026-10-19 (steps 0–7, §1.10): no verdict changed against
the tag `version-1`. The `/csdp` sweep's fixes (step 8) were made 2026-10-20 and pushed the same day with the owner's
approval; CI run 39449517708 passed on Python 3.11, 3.12 and 3.13 (S1.10c). A cold sweep's fixes were made
2026-10-20 (§1.10) and pushed the same day with the owner's approval (`0d86b29`); CI run 39482917825 passed on
Python 3.11, 3.12 and 3.13.** Written from
`docs/version_2_plan.md` §3 (stage 1) and the `TODO.md` entries it collects, each checked against the code on
2026-10-19. Later stages add their own sections here, as the Phase D plan did.

**Stage 2 plan DRAFTED 2026-10-23 (§2), awaiting the owner's decisions D8–D14.** Written from
`docs/version_2_plan.md` §3 (stage 2, items 2.1–2.5), §4 (W1–W5) and §6, `docs/spec_rules/M17.md` and `TODO.md`
T-018, each statement checked against the tree at `63409c1` (engine hash `91f9a81756a5`) on 2026-10-23. Plan only:
nothing of it is built, no record or pilot was run, and no engine or config file changed.

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

## 2. Stage 2: measurement (Phase E core)

Stage 2's five items (2.1–2.5) are listed in `docs/version_2_plan.md` §3. Stage 2 runs the constant sweep `M17.E.1`
asks for, scoped by the owner's Q3 (per criterion, the constants its mechanism reads), after a timed pilot, and
records the result beside version 1's verdicts. No rule changes in stage 2 except the detection-limit rule (D8), and no
run in stage 2 moves the engine hash (§2.6). This section was written on 2026-10-23, every statement checked against
the tree at `63409c1` (`main`, engine hash `91f9a81756a5`), and stops at the plan: nothing here is built until the
owner decides D8–D14.

### 2.1 What the code shows (checked 2026-10-23)

1. **Of `M17`'s 28 requirements, four have code and tests; the sweep has neither under `src/` nor a test.** By `docs/spec_index.jsonl` and
   `docs/spec_coverage.md` (rows 577–605): `M17.A.1` (the adaptive stopping rule; five tests in
   `tests/bowen/test_ensemble.py`), `M17.A.4` (the minimum margin) and `M17.D.3` (the arm-blind engine) are **done**;
   `M17.A.3` (the paired statistic) is **partial** — the statistic and Holm's step are tested, but no test would fail
   if the runner used an unpaired test. The other 24, `M17.E.1` among them, are **not done**, "no test named for it;
   Phase E builds it and is not built yet". `M17.A.2`, `M17.B.1`, `M17.E.3` and `M17.E.6` stay out until the pilot
   reports (Q3, decided 2026-10-19); nothing in this plan builds them.
2. **A sweep tool exists, and it is not `M17.E.1`'s.** `tools/sweep_record.py` (Phase C plan D9) runs the composite
   criteria of `src/bowen/ensemble/criteria.py` — Phase C's only — at half and double the central value of three
   learner constants (`learning_rate`, `credit_horizon`, `policy_temperature`), one constant at a time, and renders
   `docs/phase_c_sweep_record.md` from `docs/records_cache/sweep.json`. It is kept current by
   `tests/bowen/test_ensemble_record.py::test_d9_sweep_record_is_current`, which requires exactly those settings, so
   the stage 2 sweep is a new tool, not a change to this one. What it shows is how a setting reaches a run: each setting
   is a `Mutant` (`tools/mutant_runner.py:28`) that edits one row of `config/bowen/constants.md` in a temporary copy of
   the repository, run through `run_cached` (`tools/record_cache.py:146`) and keyed by the engine hash and the edit.
   `Mutant.also` carries further edits, so a joint sample of several constants is representable with the machinery as
   it stands.
3. **A run reads its constants from the file only.** `load_constants()` reads `config/bowen/constants.md`
   (`src/bowen/scenario/constants.py`), and the criterion modules bind it at import: `criteria_d.py:111` (`YEAR`) and
   `criteria_d.py:183` (`ENGINE`); `criteria.py` calls it at lines 304, 371 and 509. No runner-level override exists, so
   a sampled setting must be a file edit in a copy, as item 2 does. That is also why the sweep leaves the working tree
   and the engine hash untouched: the copy is edited, never the tree.
4. **The engine hash covers every file under `config/bowen/` and the runner.** `code_hash`
   (`tools/ensemble_record.py:38–61`) hashes `src/bowen/**/*.py` outside `render/`, every `config/bowen/**/*.md` except
   `constants_changes.md`, and the record's own tools. Two consequences for stage 2: the file that declares the sweep's
   ranges cannot live under `config/bowen/` without moving the hash (D11 puts it in `config/sweep_ranges.md`); and the
   detection-limit rule, built in `src/bowen/ensemble/runner.py` with its constants in `constants.md`, moves the hash and
   makes every record stale, so it belongs with group G's regeneration (D8, D14), not with the sweep.
5. **The population for W4 is 13 rows.** `docs/version_1_status.md` has 60 rows. `tools/version_1_status.py` calls
   composite and mixed criteria *directional* (`DIRECTIONAL`, line 67). 13 directional rows pass:
   `M11.C.2[twelve-person family]`, `M11.C.7[joint over one spouse]`, `M11.C.7[joint over sided]`,
   `M11.C.10[family of origin]`, `M11.C.11`, `M11.C.15`, `M11.C.16`, `M11.C.17`, `M11.C.22`,
   `M11.C.27[unstable,remove_one]`, `M11.C.39`, `M11.C.41[heavy: lower level]`, `M11.C.41[light: lower level]`. 19
   directional rows fail and are accepted findings. The 27 premise and check rows are not swept: a premise passing
   confirms the code and is never a finding (`M11.5`), and W4 names passing directional criteria only. One passing row
   runs a fixed seed count (`M11.C.10`, 200 seeds, `Criterion.seeds`); the runner honours it (`runner.py:115–122`), and
   so must the sweep and D8.
6. **The grades.** Every one of the 177 rows of `config/bowen/constants.md` is `[I]`. `docs/model_explainer.md` names 7
   of the 177 keys (grep of each key in backticks): `defence_threshold` (line 1276), `belief_rate` (1853) and
   `hold_gain` (1276, 2083) on lines that carry `[I]`; `practice_window` (2019), `system_adjust_gain` (275) and
   `constitution_strength` (447) on lines with no grade tag; the rest not at all. The explainer grades mechanisms, not
   keys. So the grade of every swept constant is the register's `[I]`, and for the 174 the explainer does not grade by
   name, `[I]` until proven otherwise. The explainer's §17.3 says Phase E "sweeps them all (`M17.E.1`)", which the
   owner's Q3 narrowed to per criterion; that sentence is updated when the sweep is built (§2.7).
7. **No wall time is recorded anywhere.** `tools/record_cache.py`, `tools/mutant_runner.py` and
   `tools/child_process.py` record outcomes, never durations (no `time`, `perf_counter` or `elapsed` in any of them).
   The pilot's timer is new, and its measurement is a separate page, not a cached result (S2.8): a time is not
   reproducible, so it must not be in a record a test regenerates.
8. **The status tool is frozen.** `tools/version_1_status.py` refuses to write (its docstring; owner 2026-10-19,
   stage 0.1); only `dispositions()` is live, imported by `tools/spec_coverage.py`. Its record parsing
   (`record_rows`, `red_mutants`, `not_built`, `proved_by_test`, `premise_limbs`) is importable and reusable. The version
   2 page is a sibling tool (D13), as `docs/version_2_plan.md` §4 says.
9. **The detection-limit item.** `TODO.md` T-018 (`docs/todo/2026-10-17_revision_32.md`) says `M11.C.22`'s mild limb
   has about a 78% pass chance at the runner's stop and proposes a fixed seed count when the effect is within a set
   multiple of the precision. The figure's derivation is not in that item, in `docs/version_1_dispositions.md`, in
   `docs/spec_revision_log.md` or in the changelog archive (grep "78%"); this plan takes it as the item states it. The
   runner's stop (`runner.py::_converged`) is relative: a gating readout has converged when the half-width of the
   per-seed difference is below `ensemble_precision` (0.25) times the pooled seed-to-seed sd of the two arms. The stop
   does not look at the effect's size, so an effect of about one half-width stops as early as a large one.

**What the pilot and the sweep need that does not exist:** a declared-range file and its strict parser; a per-criterion
constant set, generated and reviewed; a seeded sampler over the joint ranges; the sample-as-edits runner and the
fraction readout; a sweep record and its cache; the pilot's timer and its page; the version 2 status tool, page and
dispositions file; the detection-limit rule in the runner; and the tests of §2.3. Everything else — the copy-and-edit
child, the cache, the record-JSON parser, the criteria and their verdict rule — exists.

### 2.2 Decisions for the owner before code

Stage 1's decisions were D1–D7. Each decision below gives the choice, Claude's recommendation and the reason. Every
number proposed here is `[I]`: a modelling or measurement choice with no source.

**D8 — The detection-limit rule (version 2 plan item 2.3; changes `M17.A.1` for every criterion).**
- (a) **A fixed count near the limit.** At convergence, if any gating readout's |mean difference| is below
  `detection_limit_multiple` × its precision (the precision being `ensemble_precision` × the pooled sd, the quantity
  the stop compares the half-width with), the runner keeps adding blocks to `detection_limit_seeds`, not past
  `ensemble_cap`, and the verdict row carries `near_limit: true` and the seed count used. A criterion with a declared
  `Criterion.seeds` is unchanged. Both constants are `[I]` ensemble rules in `config/bowen/constants.md` and in
  `RULE_KEYS`; proposed `detection_limit_multiple` 2 and `detection_limit_seeds` equal to `ensemble_cap` (no second
  count constant; the cap already bounds every run). `M17.A.1` is amended to say so; every ensemble record's "Rules"
  line prints the two.
- (b) **A flag only.** The renderers mark a row `near the detection limit` when |mean difference| < 2 × half-width,
  computed from the row's JSON at render time. No extra seeds, no runner change, no hash move, no spec change beyond a
  note.
- (c) **A tighter precision for all**: `ensemble_precision` lowered. Every criterion runs longer, whether or not it is
  near the limit.

*Recommended: (a), built inside group G's revision and regeneration (D14).* T-018's case is a marginal verdict decided
by where the stop happened to land; `M13.4` adopted adaptive stopping on the owner's decision of 2026-09-22 "that a
fixed count lets ensemble size, not the mechanism, decide marginal results", and a stop that ignores the effect's size
leaves the same door open from the other side. (b) names the problem without changing the verdict, and (c) spends
seeds where none are needed. (a) moves the engine hash (`runner.py` and
`constants.md` are both in it, §2.1 item 4), so it is built in the same batch as group G and the event sizes, whose
regeneration is already "one full, shared" (`docs/version_2_plan.md` §6); then the sweep and the pilot run on a hash
that holds still. The extra seeds' cost is measured by that regeneration, not estimated here. Because the pilot runs
after D8 is built, the pilot's wall time includes D8's effect. The spec revision carrying D8 is group G's (its number
is not fixed here; `_STATUS.md` says group G's revision follows the G2 proposal).

**D9 — The pilot's two subjects (item 2.2): one passing criterion and one accepted finding.**

Chosen from `docs/version_1_status.md` (grep `^| \`M11.C`), among the 32 directional rows (§2.1 item 5).
- Passing: **`M11.C.17`** (composite, Phase D part I, PASS, `coached_rise` +0.000888 ± 0.0013, p 0.0148; the owner's
  note of 2026-10-19 says it passes "weakly"). It is the passing row with the smallest effect against its half-width,
  so a narrow band of constants is most likely to show here; its mechanism is a coach's contact that lands
  (`M1.E.7`), and its one recorded red mutant edits a register row (`perspective_gain` 0.1 → 0.0,
  `tools/phase_d_mutation_record.py:90–93`), so the sweep's range for that constant includes the mutant's value at one
  end. It runs the Phase C family for 520 weeks (`config/bowen/criteria.md:147`), the expensive class of run, which the
  pilot must time.
- Finding: **`M11.C.5`** (composite, Phase C, FAIL, accepted 2026-10-17: "No settled change-back reaction over 60
  weeks"). It is in group B, which the owner left as an accepted finding (Q2), so its sweep fraction is final for
  version 2 rather than superseded by a stage 3 proposal; and it is in the D9 record at six settings
  (`docs/phase_c_sweep_record.md`), FAIL at every one, which gives the pilot's fraction a comparison that costs nothing.
  Its mechanism is the `I-POSITION` sequence and the life-energy debit (`M5.D`, `M5.E`; spec row: "the life-energy
  debit, `M5.E.7`"). It runs a triad for weeks declared in `criteria.md`, the cheap class.
- Alternatives, if the owner prefers: `M11.C.16` for the passing subject (Phase C, in the D9 record, PASS at every
  setting; cheaper, and less likely to show anything); `M11.C.44` for the finding (its disposition says it "has moved
  between pass and fail on unrelated revisions with both mutants surviving", which a fraction over the range measures
  directly; but it is in group A, which stage 3 revisits, so its stage 2 fraction would be re-measured after any rule
  change).
- The pilot's sample count: **10 samples per subject** (`[I]`): the fraction then has a resolution of 0.1, and ten
  timings give a median per class of run. The pilot's samples are the first ten of the full sweep's sample sequence
  (same sampler seed, D12), so the full sweep reuses them from the cache.

*Recommended: `M11.C.17` and `M11.C.5`, 10 samples each.* One Phase D family run and one Phase C triad run time both
classes the full sweep will meet.

**D10 — The sweep's scope (item 2.1, Q3): which constants each criterion's mechanism reads, and how the set is made.**
- (a) **A generated set per criterion, reviewed by the owner before the pilot.** `tools/sweep_scope.py` writes
  `docs/sweep_scope.md`, one section per directional criterion, listing the `[I]` register constants whose `spec`
  column (`constants.md`'s fifth column) names a rule in the criterion's mechanism. The mechanism's rules come from
  three places, each cited in the page: the criterion's spec row (its mutation clause names the mechanism, for example
  `M11.C.44`'s "`M4.C.1`'s two-sided shape"), the `Spec:` lines of the arm's module and functions, and the register rows
  that the criterion's recorded red mutants edit (`perspective_gain` for `M11.C.17`, `lock_in_gain` for `M11.C.22`).
  The learner's constants (`M4.D.6`, `M4.D.6b`, `M4.D.6e`, `M4.D.1`) are in every composite's set, because a composite
  is by `M11.5` an outcome that is learned. A criterion whose set is empty is listed as **not covered**, never omitted
  (the form `M17.E.2` requires for its dominant constant). The owner reviews the page before the pilot (Q3's "reviewed
  by the owner before any run").
- (b) **By hand, in this plan**, for all 13 passing rows now. Not recommended: the mapping is a reading of 13
  spec rows and about 30 module docstrings, and a hand table goes stale; (a) is checked by a test and regenerated.
- (c) **Every `[I]` constant for every criterion.** Ruled out by Q3.

The two pilot subjects' sets, made by (a)'s rule by hand here so the owner sees the shape (register rows whose `spec`
column names the rule; central values as the register has them; the range by D11's rule):

| `M11.C.17` — coaching | rule | central | range (D11) |
|---|---|---|---|
| `landing_rate` | `M1.E.7` | 0.15 | 0.075–0.3 |
| `perspective_gain` | `M1.E.7` | 0.1 | 0.05–0.2 (the mutant's 0.0 is the null end of `M17.E.2`, outside this range) |
| `binder_failure_fraction` | `M1.E.7` | 0.8 | 0.4–1 (capped at 1) |
| `self_practice_gain` | `M1.E.7` | 0.001 | 0.0005–0.002 |
| `practice_window` | `M1.E.7` | 4 | 2–8 |
| `system_adjust_gain` | `M1.E.7` | 1.0 | 0.5–2.0 |
| `delayed_view_bonus` | `M1.E.7c` | 1.0 | 0.5–2.0 |
| `session_interval_weeks`, `contact_optimum`, `contact_window`, `contact_excess_exponent` | `M1.E.8` | 8, 4, 26, 2.0 | 4–16, 2–8, 13–52, 1.0–4.0 |
| `delayed_view_weeks` | `M16.D.2` | 26 | 13–52 |
| `estimator_window`, `estimator_breadth`, `estimator_sd_weight`, `estimator_rate` | `M1.A.4k` (the spec row names `estimator_breadth`) | 3, 3, 1.0, 0.5 | 2–6, 2–6, 0.5–2.0, 0.25–1 |
| `perspective_channel_gain`, `perspective_steer_gain`, `self_channel_exponent` | `M4.D.1a` | 0.5, 0.5, 2.0 | 0.25–1, 0.25–1, 1.0–4.0 |
| `learning_rate`, `credit_discount`, `soothing_review_every`, `soothing_review_penalty`; `credit_horizon`; `cross_person_weight`; `policy_temperature`, `policy_intensity` | `M4.D.6`; `M4.D.6b`; `M4.D.6e`; `M4.D.1` | 0.2, 0.7, 12, 0.5; 3; 0.5; 1.0, 100.0 | 0.1–0.4, 0.35–1, 6–24, 0.25–1; 2–6; 0.25–1; 0.5–2.0, 50–200 |

`estimator_breadth`'s range needs a check before the run: `M11.C.17`'s arms deliver nodal load "of at least
`estimator_breadth` kinds in every window" through `criteria.md`'s `nodal` settings (three kinds), so a sample at
breadth 6 can never meet the estimator's test in either arm. The scope page says which constants interact with the
criterion's own settings that way; the owner decides whether such a constant is swept over a range the settings can
meet (here 2–3) or left out.

| `M11.C.5` — change-back reaction | rule | central | range (D11) |
|---|---|---|---|
| `debit_gain` | `M5.E.7` (the spec row's mechanism) | 0.3 | 0.15–0.6 |
| `respect_gain` | `M5.E.8` | 0.2 | 0.1–0.4 |
| `opposition_window` | `M5.E.3` | 3 | 2–6 |
| `prepare_ticks`, `hold_window`, `hold_gain`, `anger_threshold`, `stall_limit`, `pull_up_rate`, `exchange_gain` | `M5.D.2a`, `M5.D.2`, `M5.D.3`, `M5.D.4`, `M5.D.4`, `M5.D.5`, `M5.D.7a` | 4, 6, 0.3, 0.3, 4, 0.2, 0.5 | 2–8, 3–12, 0.15–0.6, 0.15–0.6, 2–8, 0.1–0.4, 0.25–1 |
| `symptom_leak_rate`, `symptom_threshold_gain`, `symptom_rearm_fraction` (the third person's symptom) | `M4.C.3a`, `M7.D.1`, `M7.D.1` | 0.1, 1.5, 0.5 | 0.05–0.2, 0.75–3.0, 0.25–1 |
| the learner's constants, as above | `M4.D.6`, `M4.D.6b`, `M4.D.6e`, `M4.D.1` | as above | as above |

*Recommended: (a).* The two tables above are the shape of the page, not its final content: the page is generated from
the register and the spec rows, and the owner's review of it is the gate (§2.8).

Also under D10, which rows the full sweep covers:
- (i) the 13 passing directional rows (W4's population) — required;
- (ii) the accepted findings of the groups the owner chose for stage 3 (A, C, F: `M11.C.42`, `M11.C.44`, `M11.C.45`,
  `M11.C.27[stable,add_third]`, `M11.C.27[unstable,add_third]`, `M11.C.2`, `M11.C.6`, `M11.C.21`, `M11.C.29`; nine
  rows), so each stage 3 proposal knows whether its finding holds across the range or only at the central point;
- (iii) all 19 accepted findings.

*Recommended: (i) now; (ii) decided with N and F after the pilot, when the cost per row is known; (iii) not in stage 2.*

**D11 — The declared ranges, where they are declared, and the named ratio (item 2.1).**
- The range rule, one rule for every constant so that no range is chosen after seeing a result: **half to double the
  central value**, the D9 record's precedent, sampled log-uniformly so the central value is the median. An integer
  constant is rounded to distinct integers within that span, never below 1. A share, rate, probability or fraction
  bounded above by 1 is capped at 1; a threshold on a [0, 1] axis likewise. A constant that both arms' settings
  constrain (the `estimator_breadth` case above) is listed on the scope page with the constraint, for the owner.
- **Not swept, by name, with the reason:** `fast_tick_weeks` and `slow_tick_fast_ticks` (the clocks; every rate is per
  tick); `invariant_tolerance` and `functional_level_floor` (numerical guards); `bond_energy_decay_rate` (0.0;
  `M1.B.4` requires decay at or near zero, and half to double of 0 is 0); `death_transfer_share` (1.0; "1 is the rule",
  only `M11.C.37`'s mutant sets it lower); the seven ensemble rules (`ensemble_block` to `equivalence_margin`; they are
  the measurement, not the model) and D8's two; the life-course ages the owner set (`acting_age_years`,
  `development_start_age_years`, `development_end_age_years`, `launch_age_years`, `working_age_years`,
  `coach_start_age_years`, `coach_retirement_age_years`, `chronic_anxiety_fixation_age_years`) and
  `spouse_basic_level_tolerance`, which are initial conditions and belong to `M17.C.1`'s declared distribution, not to
  `M17.E.1` (explainer §17.3: initial conditions are what an intervention acts on, not nuisance parameters). Every
  other register constant is sweepable when a criterion's set names it.
- **Where declared:** `config/sweep_ranges.md`, a strictly parsed table (key, central as in the register, low, high,
  grade, reason-or-exclusion), outside `config/bowen/` so it is outside the engine hash (§2.1 item 4), and read only
  by the sweep tools. A row whose central value differs from the register's raises (the D9 tool's own check,
  `sweep_record.py::row_mutant`); a key not in the register raises; a range that does not contain the central value
  raises. Its header carries the sampler seed, N and F once they are decided (D12, D13). Each sample's edits are its
  cache key, so a changed range reruns only the samples it changes.
- **The named ratio.** `M17.E.1` says the rate at which ties harden or attenuate, relative to the rate at which
  anxiety moves, "MUST be swept as a named ratio". The register has `balance_harden_rate` 0.02 (`M1.B.6`),
  `tie_wear_rate` 0.05 (`M7.E.4`) and `acute_decay_rate` 0.2 (`M1.A.8`); no ratio is named in `M10`. Options: (a) the
  sweep record reports, per sample, `balance_harden_rate / acute_decay_rate` and `tie_wear_rate / acute_decay_rate`
  computed from the sampled values, and the fraction holding is shown by ratio tercile, with no new constant; (b) a
  named ratio constant added to `M10` and `constants.md`, which moves the hash and needs a revision. *Recommended: (a)
  in stage 2; (b) offered to the owner for group G's revision if the ratio is to be a register entry.*

**D12 — What one sample runs, and the sampler.**
- (a) **The runner's own verdict per sample.** Each sample runs the criterion through `run_criterion` as the records do:
  the same adaptive stopping (`M17.A.1` with D8), the same margin, Holm correction and seed keying, so "the direction
  holds in this sample" means exactly what PASS means in the record. UNDETERMINED is counted apart, never as holding.
  The sampler is a seeded Latin hypercube over the criterion's set on the log scale (`[I]`; the seed is declared in
  `config/sweep_ranges.md`), N samples per criterion, N decided after the pilot and recorded there before the full
  sweep (Q4's form).
- (b) **A fixed small seed count per sample**, cheaper and not the record's verdict rule.

*Recommended: (a).* W4 is "the fraction of samples in which its direction holds", and a direction holding is the
runner's verdict; a weaker rule inside the sweep would make the fraction incomparable with the verdict beside it. If
the pilot shows (a) is too costly, the owner's lever is N, not the rule.

**D13 — The sweep record, the version 2 status page and the "narrow" mark (item 2.4, Q4, W1, W2, W4).**
- **The record:** `docs/version_2_sweep_record.md`, generated by `tools/constant_sweep.py` from
  `docs/records_cache/constant_sweep.json`, hash-stamped like the other records. One row per swept criterion: the
  verdict at the tag `version-1`; the current record's verdict; the constants swept (count; the set is in
  `docs/sweep_scope.md`); N; the fraction of samples holding; the fraction UNDETERMINED; seeds per sample (min–max);
  the ratio terciles of D11; and **narrow** when the fraction is below F. A machine-readable block (`record_json`)
  carries every sample's values and verdict. A premise or check row never appears; the tool refuses one.
- **The status page:** `docs/version_2_status.md`, generated by `tools/version_2_status.py`, a sibling of the frozen
  version 1 tool that imports its record parsing. Columns: Criterion | Class | Version 1 (from
  `git show version-1:docs/version_1_status.md`, the frozen page) | Record | Verdict | Sweep (fraction, N; narrow) |
  Status | Disposition. The page states W1 and W2 as the version 1 page states V1–V3, and W4 as a count: "n of the 13
  passing directional rows have a sweep result". Dispositions for rows whose verdict is unchanged come from
  `docs/version_1_dispositions.md` at the tag; a new `docs/version_2_dispositions.md` (header and rules only in stage
  2) takes stage 3's decisions and any row whose verdict changes.
- **F, the narrow fraction:** decided by the owner after the pilot, from the two measured fractions, and written into
  `config/sweep_ranges.md`'s header before the full sweep runs (Q4). The tool refuses to run the full sweep without it
  and refuses to mark narrow in the pilot's report. "Narrow" is a mark on a pass, not a verdict: the row's status stays
  "pass".

*Recommended as described.* The version 1 column is read from the tag, never from the working tree, so a regenerated
record cannot rewrite what version 1 said (the same rule as `tests/test_version_1_status.py`).

**D14 — The order relative to group G (version 2 plan §6).**
- (a) **Design now, pilot after group G.** Steps 1–3 of §2.4 (the range file, the scope page, the sweep tool, the
  status tool, their tests) are built on the current hash: none of them is in the engine hash, and their tests run
  stub criteria, not the engine. D8 is built inside group G's revision. The pilot runs on the hash group G's
  regeneration leaves, so its samples are not thrown away by a hash move.
- (b) **Pilot now**, on `91f9a81756a5`, and again after group G.

*Recommended: (a), as the owner decided 2026-10-19 (`docs/version_2_plan.md` §5 Q2 and §6).* (b) measures the time
twice; the first measurement is on a runner without D8, which the second has.

### 2.3 Criteria and their tests

Each test name carries the criterion's ID and a spec ID where one applies. A test counts as coverage only once a
mutation on a copy has shown it failing; the mutation is listed. Acceptance tests in this repository assert a
direction between two arms, never a threshold; the sweep's fraction and the narrow mark are measurements beside a
verdict, and no test below asserts a fraction's value. The table is written for the recommended answers; another
answer changes its row.

| ID | Criterion | Test | Proved failing by |
|---|---|---|---|
| S2.1 (`M17.A.1` as amended by D8) | An effect within the multiple of the precision runs on to the fixed count; a clear effect stops at convergence; the row records `near_limit` and the seeds used; a declared `Criterion.seeds` is unchanged | `tests/bowen/test_ensemble.py::test_m17a1_s2_1_an_effect_near_the_limit_runs_to_the_fixed_count`, `::test_m17a1_s2_1_a_clear_effect_stops_at_convergence`, `::test_m17a1_s2_1_a_fixed_seed_criterion_is_unchanged` (stub arms with declared spreads, as the existing `test_m17a1_*` tests) | the near-limit branch removed; the multiple set to 0 |
| S2.2 (`M10.B.2`) | `config/sweep_ranges.md` is parsed strictly: a key not in the register, a central value that differs from the register's, a range that excludes the central value, or a range on a constant the file lists as not swept, raises | `tests/test_constant_sweep.py::test_s2_2_ranges_parse_strictly` (four cases) | each check removed |
| S2.3 (`M17.E.1`) | `docs/sweep_scope.md` is current, lists every directional criterion once, lists no premise or check, and marks an empty set **not covered** | `tests/test_constant_sweep.py::test_s2_3_scope_page_is_current`, `::test_s2_3_every_directional_criterion_has_a_set_or_is_not_covered`, `::test_s2_3_no_premise_is_in_scope` | a hand edit to the page; a directional row deleted; a premise row added |
| S2.4 | The sampler is seeded and reproducible, draws N samples inside every range, and the pilot's samples are the first N_pilot of the sequence | `tests/test_constant_sweep.py::test_s2_4_sampler_is_seeded_and_in_range`, `::test_s2_4_pilot_samples_are_a_prefix_of_the_full_sequence` | the seed ignored; a range bound exceeded |
| S2.5 | One sample is one `Mutant` of row edits on `constants.md` in a copy; its definition is its cache key; after a run the working tree's `constants.md` and the engine hash are unchanged | `tests/test_constant_sweep.py::test_s2_5_a_sample_is_row_edits_in_a_copy`, `::test_s2_5_the_tree_and_the_hash_are_unchanged` (a stub child) | the edit applied to the tree |
| S2.6 (`M17.E.1`, W4) | The fraction counts PASS samples only; UNDETERMINED is reported apart; the ratio terciles are computed from the sampled values | `tests/test_constant_sweep.py::test_s2_6_fraction_counts_pass_only`, `::test_s2_6_undetermined_is_reported_apart`, `::test_s2_6_ratio_terciles` | a FAIL or UNDETERMINED counted as holding |
| S2.7 (`M11.5`) | A premise or check criterion is refused by the sweep tool | `tests/test_constant_sweep.py::test_s2_7_a_premise_is_refused` | the class check removed |
| S2.8 | The pilot's page `docs/version_2_sweep_pilot.md` names the two subjects, N_pilot, the engine hash, the machine and core count, and a wall time per sample; it is written once and not regenerated by a test | `tests/test_constant_sweep.py::test_s2_8_pilot_page_names_its_subjects_and_hash` (fields present, hash matches the sweep record's) | a field removed |
| S2.9 (W4) | The sweep record is current: rendered from the cache equals the committed file; a missing result is "stale", not a rerun | `tests/test_constant_sweep.py::test_s2_9_sweep_record_is_current` (the D9 test's form) | a hand edit |
| S2.10 (W4, Q4) | The narrow mark is set only when F is declared and the fraction is below it; the full sweep refuses to run without F | `tests/test_constant_sweep.py::test_s2_10_narrow_needs_a_declared_fraction`, `::test_s2_10_full_sweep_refuses_without_f` | the refusal removed; the comparison inverted |
| S2.11 (W1, W2) | `docs/version_2_status.md` is current; every criterion has a final status or the page says which lack one; every premise and check passes or the page says which fails | `tests/test_version_2_status.py::test_w1_s2_11_page_is_current`, `::test_w1_s2_11_every_criterion_has_a_status_or_is_named`, `::test_w2_s2_11_a_failing_premise_is_named` | a hand edit; a row's status blanked on a copy; a premise row's verdict flipped on a copy |
| S2.12 (W4) | The version 1 column equals the tag's page, row by row; the sweep column shows the record's fraction and mark, "—" for premise and check rows | `tests/test_version_2_status.py::test_w4_s2_12_version_1_column_reads_the_tag`, `::test_w4_s2_12_sweep_column_matches_the_record`, `::test_s2_12_premise_rows_carry_no_fraction` | the working tree's page read instead of the tag; a premise given a fraction on a copy |
| S2.13 (W4) | Every passing directional row has a sweep result — **the version 2 gate for W4; added with the full sweep's commit (§2.4 step 6), not before, so the suite is never red by design** | `tests/test_version_2_status.py::test_w4_s2_13_every_passing_directional_row_has_a_sweep_result` | a row's fraction removed on a copy |
| S2.14 (W6) | The documents of §2.7 agree with the code | `tests/test_doc_checklist.py`, `tests/test_spec_consistency.py`, `tests/test_doc_claims.py`, the record-current tests | — |
| S2.15 (W7) | The suite passes, and CI passes on 3.11, 3.12 and 3.13 | `python3 -m pytest tests/`; `.github/workflows/tests.yml` on the pushed commit | — |

W3 (every passing composite has a recorded mutation) is carried by the existing `test_m111d_*` tests and is not
stage 2's. W5 (the owner's sign-off on each stage 3 proposal) and W8 (no output about a real family) are not
code-testable (§2.8).

### 2.4 Order of work

| Step | Work | Depends on | Moves the hash | Regeneration |
|---|---|---|---|---|
| 0 | D8–D14 decided by the owner | this plan | no | — |
| 1 | `config/sweep_ranges.md` and its parser (S2.2); `tools/sweep_scope.py` and `docs/sweep_scope.md` (S2.3); the owner reviews the sets and ranges | D10, D11 | no | none |
| 2 | `tools/constant_sweep.py`: sampler, sample-as-edits, fraction, renderer, pilot timer (S2.4–S2.10) | D12, D13 | no | none |
| 3 | `tools/version_2_status.py`, `docs/version_2_status.md`, `docs/version_2_dispositions.md` (S2.11, S2.12) | D13 | no | render only |
| 4 | D8 in `runner.py` and `constants.md`, with `M17.A.1` amended — **inside group G's revision and build** (S2.1) | D8, the G2 proposal, group G's plan | yes, with group G | one full, shared with group G |
| 5 | **The pilot**: two subjects × N_pilot samples, timed; `docs/version_2_sweep_pilot.md`; the wall time reported to the owner; N and F decided and written to `config/sweep_ranges.md` (Q4); D10 (ii) decided | steps 1–4, group G regenerated | no | pilot only |
| 6 | **The full sweep** over D10's rows; `docs/version_2_sweep_record.md`; the status page's column; S2.13 added | step 5, N, F | no | the sweep |
| 7 | One `learning-qa` pass and one correctness pass over the whole diff (before step 5, for steps 1–3; after step 6, for the record and page); every document in §2.7; `/csdp` | 6 | no | — |
| 8 | The end-of-stage-2 milestone review (`docs/milestone_review_brief.md` §2 item 1), a separate session on its own branch; it also covers group G and the event sizes (owner 2026-10-21) | 7 | — | — |

The pilot's wall time is reported to the owner before the full sweep runs, and no cost for step 6 is estimated before
that measurement (`docs/version_2_plan.md` item 2.2). Steps 1–3 can be built now, in parallel with the G2 proposal,
as §6 of the version 2 plan allows.

### 2.5 Compute

- Steps 1–3 run tests only (stub criteria), on any machine.
- Step 4 is group G's regeneration, planned there; D8 adds seeds to rows near the limit, measured by that
  regeneration.
- Step 5, the pilot: 2 subjects × 10 samples × one adaptive ensemble each, on the machine that generates the records
  (the owner's Mac, 10 cores, per §1.5 and the version 2 plan's §6). Wall time per sample and per class of run
  (a 520-week family; a triad of the weeks `criteria.md` declares): **measured by the pilot.**
- Step 6, the full sweep: 13 rows (D10 (i)) × N samples, plus D10 (ii)'s nine if taken: cost from the pilot's per-sample
  time by class; **not estimated here.** Each sample is one child process; `run_cached` runs them with `--workers`
  as the records do, and a stopped run resumes from the cache.
- This cloud session's machine has 4 cores and is used for the suite only.

### 2.6 How a changed result is reported

No step of stage 2 moves the engine hash except step 4, which is group G's build; its changed verdicts are reported as
stage 1's were, by `tools/record_changes.py`'s page against the tag (§1.6), inside group G's plan. The sweep edits a
copy, never the tree, and S2.5 checks the hash before and after.

A sweep fraction is a column beside a verdict, never a verdict. On the status page the version 1 column is read from
the tag, the verdict column from the current record, and the sweep column from the sweep record; a row reads, for
example, "pass · 0.8 of 10 samples · narrow" and its status stays "pass". A premise or check row shows "—" in the
sweep column, and the page's legend says why: a premise passing confirms the code and is never a finding (`M11.5`),
so it has no fraction to report. An accepted finding that is swept (the pilot's `M11.C.5`; D10 (ii) if taken) shows
its fraction the same way, and its status stays "fail accepted as a finding": a fraction above zero there says the
finding depends on where in the range the constants sit, which is information for the stage 3 proposal, not a pass.
A changed fraction after a later regeneration is a changed measurement, reported with the regeneration, not a defect.

### 2.7 Documents each step updates

Per `CLAUDE.md`'s same-commit rule.

- Step 0 (this commit): this file's status line; `CHANGELOG.md`; `docs/theory/_STATUS.md`'s next line;
  `docs/REVIEWER_READING_LIST.md` §7; `docs/SPEC_REVIEW_CHECKLIST.md`'s entry for this plan; `TODO.md` T-001.
- Steps 1–3: `docs/SPEC_REVIEW_CHECKLIST.md` (rows for `config/sweep_ranges.md`, `docs/sweep_scope.md`,
  `docs/version_2_status.md`, `docs/version_2_dispositions.md`); `docs/REVIEWER_READING_LIST.md` (§5's records, §7);
  `docs/model_explainer.md` §17.3 ("sweeps them all" → the per-criterion scope, with Q3's date) and §18.7's sweep
  entry; `CHANGELOG.md`; `TODO.md` (T-018 ticked when D8 is built, not before).
- Step 4: with group G's revision: `docs/bowen_agent_model_spec_v2.md` (`M17.A.1` amended; its status line),
  `docs/spec_revision_log.md`, `config/bowen/constants.md` (two rules) and `config/bowen/constants_changes.md`,
  `docs/model_explainer.md` §18.7, the generated `docs/spec_rules/M17.md` and `docs/spec_index.jsonl`, every record,
  `docs/version_2_stage_1_changes.md`'s successor page for that regeneration.
- Step 5: `docs/version_2_sweep_pilot.md` (new; checklist row); `config/sweep_ranges.md`'s header (N, F);
  `docs/version_2_plan.md` (Q4's answer recorded beside the question); `docs/theory/_STATUS.md`.
- Step 6: `docs/version_2_sweep_record.md` (new; checklist row); `docs/version_2_status.md`; `docs/spec_coverage.md`
  (`M17.E.1`'s row, and `tools/spec_coverage_overrides.json` where it says only three constants are swept, lines 99 and
  763); `docs/readouts_phase_c.md` only if a readout's definition changes, which no step here does;
  `docs/REVIEWER_READING_LIST.md`; `CHANGELOG.md`; `docs/theory/_STATUS.md`.
- Step 7: `CLAUDE.md`'s suite count line; `README.md` if it names the records.

### 2.8 Not code-testable

- The owner's review of `docs/sweep_scope.md`: whether each criterion's constant set is its mechanism's, and whether
  each range is acceptable (Q3). Proposed review: the page, one criterion at a time, as revisions 24–30 were reviewed.
- The choice of N and F after the pilot (Q4), and whether the pilot's wall time makes the full sweep worth running;
  the owner reads `docs/version_2_sweep_pilot.md`.
- What a **narrow** mark means for a result's use (the claim grade `M17.G.1` asks for is not built in stage 2).
- W8: no output presents the model as speaking about a real family; the owner reads the record and the page.
- The milestone review at the end of stage 2 (step 8).

### 2.9 Adjacent issues found while planning, not fixed

- `M17.E.1`'s text says "the `[I]` constants", and `docs/model_explainer.md` §17.3 says Phase E "sweeps them all";
  Q3 narrowed the sweep to the constants each criterion's mechanism reads. The explainer is updated at step 1–3; the
  spec's wording is not changed by stage 2 unless the owner wants the narrowing written into `M17.E.1` with D8's
  amendment.
- `tests/test_sweep.py` tests `papers/sweep.py`, an arXiv catalogue script, not a constant sweep; its name will
  collide with the stage 2 tests in any grep for "sweep". A rename (`tests/test_papers_sweep.py`) is a one-line
  change, outside this plan.
- `M17.A.3` is partial: no test fails if the runner uses an unpaired test or skips Holm's step (coverage row 579). The
  sweep reads the runner's verdicts, so the gap reaches every fraction; a test with a stub whose paired and unpaired
  results differ would close it. Outside stage 2's scope as the version 2 plan wrote it.
- `M11.C.17`'s recorded red mutant is a constant at its null (`perspective_gain` 0.0). The sweep's range (0.05–0.2) and
  the mutant's null together are what `M17.E.2`(a) calls necessity for that constant; `M17.E.2` is not in stage 2 and
  no "dominant constant" is named for any criterion (`tools/sweep_record.py`'s docstring), but the scope page's per-
  criterion sets are the candidates `M17.E.2` would need.
- `tools/spec_coverage_overrides.json` (lines 99 and 763) says `tools/sweep_record.py` sweeps only three constants;
  true today, and stale once step 6 lands.
- `tools/mutant_runner.py::run_in_copy` times a child out at 7200 s by default. A sample of a 520-week family at D8's
  fixed count may approach it; the pilot measures, and the limit is a parameter of the call, not a constant to tune in
  advance.
- No `docs/version_2_dispositions.md` exists; W5 needs one in stage 3, and the version 2 status tool needs a file to
  read in stage 2 (D13 creates it with its header). `docs/version_1_dispositions.md`'s parser (`dispositions()`) is
  the model.
- The standards directory `~/.claude/standards/` named in `CLAUDE.md` is not present on this cloud machine, so
  `learnings.md`'s P4, P7, P9 and P23 were applied from `CLAUDE.md`'s one-line summaries of them, not from the file.
