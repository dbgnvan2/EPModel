# Phase C QA gate — learning-qa failure-pattern sweep

Date: 2026-10-08
Branch: plan-phase-b (ahead of origin/plan-phase-b by 19 commits)
Reviewer: learning-qa failure-pattern sweep (inline; delegation budget unavailable this run)

RANGE: origin/plan-phase-b..HEAD
COMMITS: 19 (06751e3 … dfe3ed9) — Phase C of the Bowen agent model: src/bowen/, tools/, tests/bowen/, config/bowen/
APPLICABLE: P4, P6, P19 (incl. its source-text corollary), P21, P22, P24, P27, P29, P37 — plus the CLAUDE.md rules (engine purity / M16.B, constants-in-config, direction-of-difference acceptance tests, mutation-proved coverage, no-real-family M11.F.9, sourced-constants-in-ledger, emergent-not-written)
CHECKED: test suite (below); the ensemble runner and paired statistic; the criteria definitions; the engine core (objects, events, tick, params, draws, event_store, sinks, contact, appraise, moves, iposition, learner, policy, readouts/counterfeit); the gate tools (ensemble_record, mutation_record, sweep_record, spec_coverage); config/bowen/constants.md; the engineering/ensemble-record/beliefs/phase-c-gate tests; docs/phase_c_completion_report.md. Greps for magic literals, engine file I/O, bare-except, and floor assertions.
NOT COVERED: a line-by-line read of every remaining mechanism module (reactive, symptoms, outside_ness, beliefs, recompute, consolidate, standing_load, visibility, act, observe, event_effects, external, invariants, log_records, state, readouts/patterns, scenario/*) — covered instead by the passing suite, the per-tick invariant assertions, the byte-identical determinism/permutation gates, and the mutation-proved magic-literal guard. Also not covered: the 19 ensemble-marked criterion runs themselves (deselected by design; their committed record is checked by the default suite), and plan §9's pending human reviews.

---

## Test result

`python3 -m pytest tests/ -q` — **478 passed, 19 deselected** (24.69 s). The 19 `ensemble`-marked criterion tests are excluded by `pytest.ini` by design and run with `python3 -m pytest -m ensemble`. Green.

## Findings

Ranked by severity. One MEDIUM, three LOW. No HIGH.

### 1. MEDIUM — P6 / P19 — the mutation and sweep records' staleness hash omits their own inputs

- Location: `tools/ensemble_record.py:36` (the `code_hash()` path list) — reused at `tools/mutation_record.py:314` and `tools/sweep_record.py:103`. Load-bearing inputs it does not cover: the `MUTANTS` definitions at `tools/mutation_record.py:85-190` and the D9 `SETTINGS` at `tools/sweep_record.py:34-38`.
- Risk: `code_hash()` walks only `src/bowen/**/*.py` and `config/bowen/**/*.md`. The mutation record and the D9 sweep record stamp that same hash, but their defining inputs live in `tools/`, which the hash does not cover. So a change to a mutant's target criteria list, to its `old`/`new` text (when the new text still matches exactly once), or to the sweep's central/low/high values leaves the committed record stale while `test_m111d_mutation_record_is_current` and `test_d9_sweep_record_is_current` stay green. The compensating tests (`test_m111a_every_mutant_applies_exactly_once`, `test_m111d_every_passing_criterion_has_a_mutant_run`) catch only two drift shapes: a mutant that no longer applies exactly once, and a passing criterion with zero mutants run. A broadened target list, a semantically changed replacement that still matches once, or changed sweep values pass silently.
- Why it matters: the "proved by mutation" claims in `docs/phase_c_mutation_record.md` and `docs/spec_coverage.md` are the currency of this project's gate, and they can drift from the actual mutant definitions with no default-suite failure.
- Fix: extend `code_hash()` to include `tools/` (at minimum `tools/mutation_record.py` and `tools/sweep_record.py`), or hash the `MUTANTS`/`SETTINGS` objects directly into their records. Re-run the two record generators afterward.
- Confidence: high. No wrong result in the tree today (the records are current and the suite is green); this is a guard-completeness gap, not a live defect.

### 2. LOW — P19 (corollary) — `not_built` is parsed from prose, not from the record's machine-readable block

- Location: `tools/spec_coverage.py:63` (regex over the ensemble record's "## Not built in Phase C" bullet list) vs the producer `tools/ensemble_record.py:93-94` (writes those bullets in prose; the ````json` block at `:100-103` omits `not_built`).
- Risk: the coverage report derives "not built" by matching the prose `- `M11.C.N` —` form. A formatting drift (e.g. the em-dash becoming a hyphen) silently empties `not_built`, dropping M11.C.7/13/14 to a generic "not done" note. `test_m134_record_covers_every_criterion_and_names_what_is_not_built` asserts the record's prose, not spec_coverage's parse of it — two contracts that can drift independently.
- Fix: emit `not_built` in the ensemble record's machine-readable JSON and have `spec_coverage.py` read it from there.
- Confidence: high.

### 3. LOW — P19 (corollary) — a bare-substring source assertion guards the belief update

- Location: `tests/bowen/test_beliefs.py:148` — `assert "true_counterpart" not in inspect.getsource(update_beliefs)`.
- Risk: this is the only guard against `update_beliefs` calling `true_counterpart` (the adjacent `true_state_reads` at `:129-134` walks the AST for `ast.Attribute` reads and cannot see a function call, which is an `ast.Name`). A rename of `true_counterpart` anywhere in `beliefs.py` silently kills the guard — the exact "dead guard" shape the corollary warns about. The adjacent check already does it correctly via `ast.walk`.
- Fix: replace the substring test with an AST call-site check (walk for `ast.Call` whose `func` is a `Name` in a declared cheat set), or fold `true_counterpart` into the parsed check.
- Confidence: high.

### 4. LOW (informational) — P4 — the criterion scenario parameters are hardcoded outside the magic-literal guard

- Location: `src/bowen/ensemble/criteria.py:68` (`SPELL_INTENSITY=120.0`, `SPELL_EVERY=4`, `SPELL_FROM=4`), `:437` (`OUTSIDE_ACTS`, `INSIDE_ACTS`), and the criterion `settings` dicts at `:456-510`.
- Risk: these are simulation-scenario parameters in Python rather than config, and the M11.D.2 magic-literal guard (`tests/bowen/test_engineering_gates.py:85`, `SCANNED = engine, policy`) deliberately does not scan `src/bowen/ensemble`, so they are unguarded. They are declared `[I]` in the module docstring (`criteria.py:10-12`), which is the honest treatment, and they are acceptance-test scenario definitions rather than engine mechanism constants — so this is defensible — but it is a deliberate unguarded region and should be recorded rather than discovered.
- Fix (optional): document why criteria settings are exempt from M11.D.2, or move the spell/readout constants to `config/bowen/` and load them.
- Confidence: high.

---

## Explicitly out of scope, and it must be stated

The Phase C acceptance gate does **not** pass. `docs/phase_c_completion_report.md:18-21` and §1 record it: 5 criteria fail (C.3, C.4, C.5, C.29, C.44), 1 is undetermined at the seed cap (C.16), 3 are not built (C.7, C.13, C.14), and several pass only in some cells or only near the frozen constants (C.27, C.41, C.42, C.45). The completion report handles this correctly: the constants are frozen (`M10.B.4`), the failures are "reported, not fixed" on the explicit ground that changing a rule to make a criterion pass would be tuning against it, and each needs an owner decision (§3, §6). That is a scientific/domain result about whether the model reproduces the spec's expected directions — not a failure-pattern defect in the code — and it is outside this sweep's scope. "APPROVED" below is the failure-pattern verdict only; it does not contradict "Phase C cannot be declared done while its gate fails."

---

## Verdict

VERDICT: APPROVED

The diff is clean against P1–P37 and the CLAUDE.md rules at high and medium-live-defect severity: the engine is pure (no file I/O; the persistence sink is a caller-supplied pure observer proved byte-identical by `test_m16t3_sink_does_not_change_results_under_the_policy`); constants are in config with per-row grades; acceptance criteria assert directions of difference over paired keyed-seed ensembles with a signed-rank test, Holm correction and adaptive stopping; the magic-literal guard is mutation-proved and exact; spec references are exact-count; mutations run in temporary copies, refuse a non-unique match, and decide from exit codes (P37/P24 resolved). One MEDIUM guard-completeness gap (finding 1) is recorded as a required fix before Phase D or before the mutation/sweep records are next regenerated; findings 2–4 are low-severity backlog items.
