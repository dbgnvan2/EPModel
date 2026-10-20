# Version 2: implementation plan

**Status: stage 1 APPROVED 2026-10-19.** The owner decided D1–D7 the same day, every one as recommended (§1.2), and
approved the plan. **Code, tests and documents built 2026-10-19 (steps 0–5, §1.10); the review, the run and
the report (steps 6–8) are next.** Written from
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
   (`exp`, `log`, `sqrt`) differs between macOS and Linux (`TODO.md`, the CI entry of 2026-10-15), so the golden
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
- `TODO.md`'s rule 9 entry (the readouts' act sets) is half done: the sets are in config, and only the engine's two
  classifications remain (S1.3).

### 1.10 Build record (2026-10-19)

Steps 0–5 are built: commits `423bbed` (steps 0–2), `b6ca902` (step 3), `36fad6c` (step 4) and step 5's documents.
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
- **S1.5a moved the engine hash at step 1.** `tools/mutant_runner.py` is in the engine hash, so the failed-child fix
  moved it one step before §1.4 expected. No record was regenerated between steps, so the run at step 7 is still the
  only one.
- **The Phase D cache's engine part is the hash the Phase D records print** (`code_hash()` with no tool), not
  `engine_hash()`, which adds `tools/mutant_runner.py`; the Phase D ensembles do not run through that tool. The key is
  otherwise built as Phase C's are (`result_key`).

**Adjacent issues found while building, not fixed** (each is in `TODO.md` or below):
- Act sets still written in the code outside the two classes S1.3 moved: `NEVER_REFUSED` (`rejection.py`),
  `ABORT_ACTS` (`iposition.py`), the sets in `sinks.py` and `moves.py`, and `NEEDS_TRIAD` in `policy.py` (rule 9).
- `tools/probes.py` still takes float means with the builtin `sum` (lines 139, 141, 268, 458). It is outside
  `src/bowen` and no verdict reads it.
- `monte_carlo_results.txt`, the grid engine's output, is still tracked.
- `tools/mutation_gate_phase_b.json` names `tests/test_simulator.py`; it is an input to the frozen Phase B record
  (D5) and is left as it was.
- `src/sim_audit.csv`, an untracked output of the retired grid engine, is the owner's to delete.
