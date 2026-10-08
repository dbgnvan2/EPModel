# Phase C QA gate (re-gate) — learning-qa failure-pattern sweep

Date: 2026-10-08
Branch: plan-phase-b (ahead of origin/plan-phase-b by 20 commits)
Reviewer: learning-qa failure-pattern sweep (inline re-gate of commit 5931aa5)

This is a re-gate. The prior gate (`docs/cycles/2026-10-08_phase-c-qa-gate.md`, APPROVED with 1 medium
and 3 low findings) reviewed 19 commits. Since then exactly one commit landed — `5931aa5` — which fixes
findings 1, 2 and 3 and defers finding 4. This pass verifies those fixes, reviews `5931aa5` in full, and
confirms nothing else in the range regressed.

RANGE: origin/plan-phase-b..HEAD
COMMITS: 20 (06751e3 … 5931aa5). New since the last gate: `5931aa5 bowen: fix the Phase C QA gate's findings; document the batch`. `origin/plan-phase-b` is unchanged at 40922e9; the previously-approved 19 commits are untouched.
APPLICABLE: P4, P6, P19 (incl. its source-text corollary and the parallel-hand-copy pitfall), P24, P27, P29, P37 — plus the CLAUDE.md rules (constants-in-config, direction-of-difference acceptance tests, mutation-proved coverage, engine purity).
CHECKED: the full `5931aa5` diff (code and docs); the three fixes read in context against their producers and consumers (`tools/ensemble_record.py`, `tools/mutation_record.py`, `tools/sweep_record.py`, `tools/spec_coverage.py`, `src/bowen/ensemble/criteria.py`, `tests/bowen/test_beliefs.py`, `tests/bowen/test_ensemble_record.py`); the regenerated records' hashes; the coverage reclassification (M11.1b/c/d) against the override JSON and generator; the default suite (below).
NOT COVERED: the 19 `ensemble`-marked criterion runs (deselected by design; their committed records are checked by the default suite), and plan §9's pending human reviews — as before.

---

## Test result

`python3 -m pytest tests/ -q` — **480 passed, 19 deselected** (24.70 s). Green. The 19 `ensemble`-marked
criterion tests are deselected by `pytest.ini` by design. This matches the count claimed in CLAUDE.md and
the completion report.

---

## Verification of the three fixes

### Finding 1 (MEDIUM, P6/P19) — hash now covers the generating tools: FIXED for its named inputs, with a residual (see finding A below)

`code_hash()` now takes `*tools` and appends them to the hashed path list (`tools/ensemble_record.py:33-48`).
`tools/mutation_record.py:43` defines `HASHED_TOOLS = (Path(__file__).resolve(),)` and writes
`code_hash(*HASHED_TOOLS)` (`:315`); `tools/sweep_record.py:34` defines `HASHED_TOOLS = (Path(__file__).resolve(), tools/mutation_record.py)`
and writes `code_hash(*HASHED_TOOLS)` (`:105`). Both records were regenerated — `docs/phase_c_mutation_record.md:10`
now carries `4c7199a…`, `docs/phase_c_sweep_record.md:10` carries `f94444ab…` — and only the hash lines changed,
so the results are unchanged as the commit claims. The mutation-record staleness test uses the single source
(`tests/bowen/test_ensemble_record.py:67`, `_tool.code_hash(*_mutants.HASHED_TOOLS)`), and a new regression test
(`test_m111d_mutation_hash_covers_the_mutant_list`) asserts the tool is hashed. Verified correct for the two
inputs the finding named (the `MUTANTS` list, the D9 `SETTINGS`). See finding A for what it misses.

### Finding 2 (LOW, P19 corollary) — not-built criteria read from the source, not the prose: FIXED

`tools/spec_coverage.py:63-66` now imports `NOT_BUILT` from `src.bowen.ensemble.criteria` instead of regex-matching
the record's prose bullets. The producer writes those bullets from the same object (`tools/ensemble_record.py:52`
imports `NOT_BUILT`; `:100` renders `not_built.items()`), so the record and the coverage report now share one
source of truth. `NOT_BUILT` (`criteria.py:512-516`) is `{M11.C.7, M11.C.13, M11.C.14}`, all matching the old
`M11\.C\.\d+` pattern, so the set is unchanged — the fix removes a drift surface without changing behaviour.

### Finding 3 (LOW, P19 corollary) — belief store guard is now an AST check: FIXED

`tests/bowen/test_beliefs.py:147-150` adds `names_used()` (walks the AST for `ast.Name` and `ast.Attribute`), and
`:153-155` asserts `true_counterpart.__name__ not in names_used(inspect.getsource(update_beliefs))`. The guard is
named through the imported function, so renaming `true_counterpart` in `beliefs.py` breaks the import loudly rather
than silently emptying a substring match. A mutation test (`:158-162`) injects `true_counterpart(tie).tension` and
confirms the guard catches it. This resolves the "dead guard" shape the finding described.

### Finding 4 (LOW, P4) — criteria literals outside M11.D.2's scan: NOT fixed, correctly deferred

Still open, recorded in `TODO.md` ("Hermes gate finding 4 … do it before Phase D"). `criteria.py:9-12` declares
the horizons/spell/readouts `[I]`. This matches the prior gate's own optional/low grading, and the deferral is
honest (the fix would change the code hash and every record).

---

## New findings (from reviewing 5931aa5 itself)

### A. MEDIUM — P6/P19 (residual of finding 1) — `RULE_KEYS` is a load-bearing tool-side input no record hashes

- Location: `tools/ensemble_record.py:29-30` (defines `RULE_KEYS`), imported by the mutation child at
  `tools/mutation_record.py:224` (`from tools.ensemble_record import RULE_KEYS`) and used by the ensemble record's
  own `run()` at `tools/ensemble_record.py:57`. The `HASHED_TOOLS` tuples (`tools/mutation_record.py:43`,
  `tools/sweep_record.py:34`) both omit `tools/ensemble_record.py`, and the ensemble record's own hash
  (`code_hash()` with no tools, `tools/ensemble_record.py:119`) never includes `tools/`.
- Risk: `RULE_KEYS` selects which `config/bowen/constants.md` values become the `rules` dict passed to
  `run_criterion`, so it is an input to every verdict. A change to it (add/rename an ensemble rule key) changes the
  mutation, sweep and ensemble verdicts without changing any record's `code_hash`, so
  `test_m111d_mutation_record_is_current`, `test_d9_sweep_record_is_current` and `test_m134_ensemble_record_is_current`
  all stay green over stale records — exactly the silent-staleness class finding 1 identified, one layer deeper. The
  fix's own docstring claims "The ensemble record's verdicts do not depend on any tool" (`tools/ensemble_record.py:38`),
  which is inaccurate: they depend on `RULE_KEYS`, which lives in the tool.
- Fix: add `tools/ensemble_record.py` to both `HASHED_TOOLS` tuples (and ideally to the ensemble record's own hash),
  then regenerate all three records.
- Confidence: high on the dependency and the class; no wrong result in the tree today (the records are current and
  the suite is green).

### B. LOW — P19 (parallel hand-copy) — the sweep staleness test copies `HASHED_TOOLS` instead of importing it

- Location: `tests/bowen/test_ensemble_record.py:89` hard-codes `(sweep_record.py, mutation_record.py)`, a second
  copy of `tools/sweep_record.py:34`'s `HASHED_TOOLS`. The mutation-record test does it right at `:67` (uses
  `_mutants.HASHED_TOOLS`); the sweep test does not.
- Risk: if `HASHED_TOOLS` changes, the record regenerates with a new hash and the test computes the old hash, failing
  loudly — but with a misleading "stale: rerun python3 tools/sweep_record.py" message that re-running cannot fix
  (the test's copy is what is stale, not the record). The drift is caught, but at the cost of a wrong diagnosis.
- Fix: load `tools/sweep_record.py` via `spec_from_file_location` (as `_tool`/`_mutants` already are) and assert
  against its `HASHED_TOOLS`.
- Confidence: high.

---

## Explicitly out of scope, unchanged from the prior gate

The Phase C acceptance gate still does not pass (5 criteria fail, 1 undetermined, 3 not built; several pass only in
some cells or near the frozen constants). That is a scientific/domain result reported in
`docs/phase_c_completion_report.md`, not a failure-pattern defect, and is outside this sweep's scope. The coverage
reclassification in this commit (M11.1b `not done`, M11.1c/M11.1d `partial`) is honest — it stops over-claiming two
criteria as `done` on the strength of tests that proved less than the spec's full text — and the count
(217 done / 44 partial / 281 not done = 542) reconciles.

---

## Verdict

VERDICT: APPROVED

The three fixes are correct and tested; finding 4 is correctly deferred; nothing in the previously-approved 19
commits changed, and the default suite is green (480 passed, 19 deselected). One MEDIUM guard-completeness gap
(finding A) remains — a residual of the same P6/P19 class finding 1 identified: `RULE_KEYS` in
`tools/ensemble_record.py` is a load-bearing input to all three records that no record hashes, and the fix's own
docstring states an inaccurate premise about it. It is a guard gap, not a live defect (no wrong result in the tree
today). It is recorded as a required fix before Phase D or before the records are next regenerated, alongside the
still-open finding 4; finding B is a low-severity backlog item.
