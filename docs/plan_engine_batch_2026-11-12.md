# Plan — the engine batch of 2026-11-12 (T-001 step 3)

*Status: approved by the owner 2026-11-12 through `AskUserQuestion`, with each choice as recommended: T-237 (b) reads the learner's own state (cross-person relief 0, the witnessed change the witness's own level change); T-219 (2) gates every couple's children by their own parents' marital tie; T-219 (11, 12) moves the act sets to `config/bowen/act_classes.md` now. Built 2026-11-12 as spec revision 59; the records are regenerated on the new engine hash at step 4.*

Step 3 of the owner's order of 2026-11-11 (`docs/todo/2026-08-22_blocking_the_next_phase.md` lines 15-23): the one
batch in that order that moves the engine hash (`d3779684dd53` now). Every rule change goes into **spec revision 59**.
Records are not regenerated here; that is step 4. Tier: Full (`src/bowen/`, `config/bowen/`). Fix batch, so the test
ceiling is one guard per finding fixed.

## Acceptance criteria and their tests

Each test is written first and seen to fail before its fix. All new tests go in the existing file named.

| # | Item | Criterion (spec ID) | Test |
|---|---|---|---|
| 1 | T-214 | `M9.8`: a person, tie or household that enters the field (pairing, birth, succession) gives every holder its own `TieBelief` object; updating one moves no other (`life_course._introduce`) | `tests/bowen/test_life_course.py::test_t214_each_belief_made_at_an_entry_is_its_own` |
| 2 | T-215 (a) | `M5.D.6`: a mover who dies in `RESOLVE` reverts the target's pull-up, logged, as a follow-up that cannot be made | `tests/bowen/test_iposition.py::test_m5d6_t215_a_mover_dead_in_resolve_reverts_the_pull_up` |
| 3 | T-215 (b) | `M5.D.3`: an abort ended by a second deferral restores the prior balance (`restore_prior_balance`, guarded for a tie that is gone) | `tests/bowen/test_iposition.py::test_m5d3_t215_a_twice_deferred_abort_restores_the_balance` |
| 4 | T-234 | `M5.D.9` (amended): while a sequence runs, `I-POSITION` toward anyone other than its target is not legal (`legal_outcomes`, via a new `Observation.sequence_target`, the person's own state); a scripted one raises in `act` (the gate at the sink too, P43) | `tests/bowen/test_iposition.py::test_m5d9_t234_no_second_iposition_while_a_sequence_runs` (as built: one test holds both checks; after the correctness review any `I-POSITION` that reaches the act while a sequence runs raises, since the tick has already made one toward the target `STAY-IN-CONTACT`) |
| 5 | T-216 | `M11.1f` (amended): the required-move gate reads each arm's own moves; `Criterion` gains `requires_moves_baseline`; `M11.C.18`, `M11.C.20`, `M11.C.23` split their tuples by arm | `tests/bowen/test_ensemble.py::test_m111f_t216_a_baseline_move_that_never_occurs_fails` |
| 6 | T-217, T-239 (c) | `M11.1f` (amended): every count an arm records (the parenthesised `moves` keys) reaches the verdict (`Verdict.counts`, per arm) and the ensemble and Phase D records; every readout carries its share of non-zero pairs (`n_nonzero / seeds`), report-only rows included. The arms of `M11.C.6` (child dead), `M11.C.25` (no pole formed) and `M11.C.26` (runs booked) record their count where they do not now. `mutant_runner.child` emits both | `tests/bowen/test_ensemble.py::test_m111f_t217_an_arms_count_reaches_the_verdict` |
| 7 | T-218 | `M16.A.1`: `config_hash` covers every loaded config input — the whole `EventKinds`, `family.life`, the policy rules — by hashing the loaded values (`plain`), so a new field cannot be missed | `tests/bowen/test_log.py::test_m16a1_t218_every_config_input_moves_the_hash` (as built: in `test_log.py`) (one structural check over the dataclasses' fields) |
| 8 | T-219 | one guard each: (1) a snap-up to the chronic floor is logged under its own name, not as positive `acute_decay`; (3) `_desc` prefers the shorter id; (4) `complete` logs the clamped gain; (5) `act` requires `params`; (6) loaders reject `nan`/`inf` and a negative `invariant_tolerance`; (7) a bad binder target fails at load with file and line; (8) the `partners:` path must name a file inside the config directory; (9) `M6.I.8`'s tie half is computed from the ties actually loaded, and proved by a mutant; `M6.I.2` recorded as a structural check that cannot fire while `Sink` has three members; (10) `fast_tick_weeks` removed from the schema, config and `M3.A.1`. **Owner questions:** (2) and (11)-(12) | as built, one per item: `test_mechanisms.py` (1), (5), (9), `test_attribution.py` (2), `test_moves.py` (3), `test_iposition.py` (4), `test_config.py` (6), (10), (11) with (12), `test_script.py` (7), `test_family.py` (8). (7)'s triangle-position binder is checked for its three members only, not for an existing triangle, since triangles form during the run |
| 9 | T-239 (d) | `M16.A.4`, `M4.G.2`: a tick whose invariants fail emits its records and an `InvariantRecord` with the failure, then raises; logging stays a pure observer | `tests/bowen/test_tick.py::test_m16a4_t239_a_failing_tick_is_logged_before_the_raise` |
| 10 | T-239 (a) | `M11.5` and explainer §19: `settle_functioning` listed among the set mechanisms (act physics, `M1.B.5`, `M1.B.6`) | document only |
| 11 | T-239 (b), T-213 (b) | `M11.5`'s rows: `M11.C.55` — the part of its reversal with learning off that holds by definition (a tie names no parent) is not evidence of learning; `M11.C.48[phase_d]` — with learning removed the pass is carried by the policy's reading of level (C1, C2, measured at `d3779684dd53`). The audit renderer prints the `M11.C.55` note | document, and `tests/test_emergence_audit.py::test_m115_t239_a_reversed_row_prints_its_m115_note` for the renderer (as built) |
| 12 | T-237 | `M4.B.2` states the learner's reads of others' true anxiety and level (`M4.D.6e`, `M4.D.6f`) as an `[I]` exemption. Two mutants, defined here and run at step 4: (a) the felt relief minus the expected decay toward the chronic floor, in `tools/mutation_record.py` (`M11.C.16`) and `tools/phase_d_mutation_record.py` (`M11.C.39`, `M11.C.34`); (b) the learner's perceived values in place of others' true state, on `M11.C.10[family of origin]`. **Owner question:** what (b) reads | `tests/bowen/test_ensemble_record.py::test_m111a_every_mutant_applies_exactly_once` (existing) |
| 13 | T-242 | `M11.D.8`, `M11.D.9` (amended): the scan covers every `.py` file under `src/bowen/` and every reference resolves; `EXPECTED_SPEC_REFERENCES` goes | `tests/bowen/test_engineering_gates.py::test_m11d8_spec_references_resolve`, `::test_m11d9_a_narrowed_scan_is_caught` (rewritten) |
| 14 | T-226 | 49 of the 60 finished documents the review lists, and the 17 Phase C QA-gate logs, move to `docs/archive/`; live references rewritten by script; kept in place: the two cited by version 1's frozen files, the two LLM-experiment documents (the prompts key its cache), and the seven read by path or still live (`milestone_review_brief.md`, `two_mac_sweep.md`, `implementation_plan_version_2.md`, and four that tools or tests read) | `tests/test_doc_checklist.py`, `tests/test_doc_claims.py` (existing). As built, each moved entry in `SPEC_REVIEW_CHECKLIST.md` keeps its line, linked into `archive/`, beside one `archive/` link, rather than being replaced by it, so the entries' descriptions stay |

Notes on corrections to the item texts: T-237 says its mutants live in `tools/mutant_runner.py`; they are defined in
the two record tools, which are outside the engine hash. `mutant_runner.py` changes only for criterion 6. T-219's
citation of `src/bowen/io/sinks.py:40-45` is wrong (no act set there); the other three act-set sites are real.

## T-158's strict expected failure

Criterion 4 removes the overwrite of a running sequence, so `test_m5d3_abort_is_the_usual_first_outcome` may change.
It is run once criterion 4 is built. If it starts to pass (the strict `xfail` turns red), the batch stops and the
owner is asked; the marker is not removed without that answer.

## Order

1, 2-3, 4 (then T-158's check), 9, 7, 8, 5-6, 13, 12, 10-11, 14 last (it rewrites paths across the tree), then the
spec revision 59 text, `docs/spec_revision_log.md`, explainer, `docs/readouts_phase_c.md`,
`config/bowen/constants_changes.md`, the generated indexes (`tools/spec_index.py`, `tools/spec_rules_view.py`,
`tools/status.py`, `tools/todo_index.py`), `docs/REVIEWER_READING_LIST.md`, `CHANGELOG.md`.

## Review and push

One full `learning-qa` pass and one correctness pass over the diff, before push. Suite run as
`nice -n 19 python3 -m pytest tests/`, exit line read before any git command. The new count is reported with its
reason. Fast-forward into `main`, push, no tag. The new engine hash is recorded in `docs/STATUS.md` and the CHANGELOG.

## What is not in this batch

T-235, T-236 and T-242's P5 disposition rewrite (after step 4); T-238, T-240 (later); T-223 and T-227 (not ordered).
