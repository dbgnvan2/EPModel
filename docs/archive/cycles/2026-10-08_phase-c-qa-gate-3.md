# Phase C QA gate (re-gate 2) — learning-qa failure-pattern sweep

Date: 2026-10-08
Branch: plan-phase-b (ahead of origin/plan-phase-b by 1 commit)
Reviewer: learning-qa failure-pattern sweep (inline re-gate of commit `ca9b311`)

This is a re-gate. The prior gate (`docs/cycles/2026-10-08_phase-c-qa-gate-2.md`, APPROVED with 1 MEDIUM
and 1 LOW finding) reviewed 20 commits. Since then exactly one commit landed — `ca9b311` — which fixes
re-gate findings A and B and the first gate's finding 4. This pass verifies those three fixes, reviews
`ca9b311` in full (including `config/bowen/criteria.md` and the refactor of
`src/bowen/ensemble/criteria.py`), and confirms nothing in the range regressed.

RANGE: origin/plan-phase-b..HEAD
COMMITS: 1 (`ca9b311 bowen: fix re-gate findings A and B and gate finding 4`). `origin/plan-phase-b` is
unchanged at `347b688`; the previously-approved 20 commits are untouched.
APPLICABLE: P4 (hardcoded constants), P6 (staleness-hash completeness), P19 (producer/consumer drift,
incl. its source-text corollary and the parallel-hand-copy pitfall) — plus the CLAUDE.md rules
(constants-in-config, direction-of-difference acceptance tests, mutation-proved coverage).
CHECKED: the full `ca9b311` diff; the three fixes read in context against their producers and consumers
(`tools/ensemble_record.py`, `tools/mutation_record.py`, `tools/sweep_record.py`,
`src/bowen/ensemble/criteria.py`, `tests/bowen/test_ensemble_record.py`,
`tests/bowen/test_engineering_gates.py`); the regenerated records' hashes; the byte-identity of the
regenerated records against their parents; a differential probe of the criteria refactor (old module exec'd
against the new module); the magic-literal scan's coverage of `criteria.py`; the default suite (below).
NOT COVERED: the 19 `ensemble`-marked criterion runs (deselected by design; their committed records are
checked by the default suite), and plan §9's pending human reviews — as before.

---

## Test result

`python3 -m pytest tests/ -q` — **483 passed, 19 deselected** (24.46 s). Green. The 19 `ensemble`-marked
criterion tests are deselected by `pytest.ini` by design. This matches the count now claimed in CLAUDE.md
and the completion report (480 + the 3 tests added by this commit).

---

## Verification of the three fixes

### Finding A (MEDIUM, P6/P19) — every record's hash now covers `RULE_KEYS`: FIXED, complete

`tools/ensemble_record.py:29` defines `HASHED_TOOLS = (Path(__file__).resolve(),)` and `:121` writes
`code_hash(*HASHED_TOOLS)`, so the ensemble record hashes its own `RULE_KEYS`.
`tools/mutation_record.py:44` sets `HASHED_TOOLS = (Path(__file__).resolve(), REPO / "tools" / "ensemble_record.py")`
and `:316` writes `code_hash(*HASHED_TOOLS)`; `tools/sweep_record.py:35` sets the same plus
`mutation_record.py`, and writes it at its own site. The dependency chain is real and complete:
`RULE_KEYS` feeds the ensemble record directly (`tools/ensemble_record.py:59`) and the mutation/sweep
verdicts through `tools/mutation_record.py:225,228` (`child()` rebuilds `rules` from `RULE_KEYS`), which
`sweep_record.py` reuses via `run_mutant`. `code_hash` resolves every tool path before hashing
(`tools/ensemble_record.py:44`, `paths += sorted(Path(t).resolve() for t in tools)`), so the relative
`REPO / "tools" / "ensemble_record.py"` forms hash identically to the ensemble tool's own resolved path —
confirmed by the differential hash below. The inaccurate docstring ("the ensemble record's verdicts do not
depend on any tool") is corrected (`tools/ensemble_record.py:37-40`). The regression test
`test_m134_every_record_hashes_the_ensemble_tool` (`tests/bowen/test_ensemble_record.py:110-115`) asserts the
tool is in all three tuples (resolved) and that hashing it changes the hash. The three records were
regenerated: their hashes changed (`f47af691…`→`5e9ce247…`, `4c7199a…`→`c72face6…`, `f94444ab…`→`8f0afcbd…`).

### Finding B (LOW, P19 parallel-hand-copy) — the sweep staleness test imports `HASHED_TOOLS`: FIXED

`tests/bowen/test_ensemble_record.py:94` now computes `_tool.code_hash(*_sweep.HASHED_TOOLS)` where `_sweep`
is loaded from `tools/sweep_record.py` via `spec_from_file_location` (`:47-55`). The hand-copied tuple
`(sweep_record.py, mutation_record.py)` is gone; there is now a single source of truth for the sweep tool's
hashed inputs.

### Finding 4 (LOW, P4) — criteria settings moved to config, unchanged: FIXED, differential-verified

`config/bowen/criteria.md` (new, 69 lines) holds every declared setting. `src/bowen/ensemble/criteria.py`
loads it via `load_settings()` (`:101-121`) with a strict parse: a row naming a setting not in `REQUIRED`
(`:69-88`) raises, a duplicate raises, and a missing required setting raises. `M11.D.2`'s magic-literal scan
now covers `criteria.py` (`tests/bowen/test_engineering_gates.py:87,102-103`), with a mutation-proved check
(`:106-112`) that a literal re-introduced into `criteria.py` is caught.

The refactor is behaviour-preserving, verified two ways:

1. **Byte-identity of the regenerated records.** `git show origin/plan-phase-b:docs/phase_c_*_record.md`
   minus the `code_hash:` line is byte-identical to the committed record for ensemble, mutation and sweep.
   Every verdict, seed count, difference, flag and mutant result is unchanged, so the move to config
   changed no run outcome.
2. **Differential probe.** Exec'ing the old module (from `origin/plan-phase-b`) and importing the new one:
   criterion ids identical (no missing on either side); every old settings key has an equal value in the
   new settings (0 mismatches); the new-only keys are exactly the moved literals (e.g.
   `M11.C.3 → {run_after: 2, latency: 1}`, `M11.C.4 → {run_after_nodal: 2}`, `M11.C.27 → {run_after: 3,
   unstable_impingement: 0.8}`, `M11.C.29 → {disguise_span: 8, disguise_every: 2}`, …); `spell()` is
   equivalent across every family and weeks (12/52/80/104); and the C.41 light-stress spell
   `old.spell("phase_c", w)[::4]` equals `new.spell("phase_c", w, every=light_every, parents=light_parents)`
   for weeks 40 and 80. No external consumer of `SPELL`, `act`, `SETTINGS` or `spell` exists (grep
   confirmed), so the removed `SPELL_INTENSITY/SPELL_EVERY/SPELL_FROM` and the `act(..., intensity=100.0)`
   default are safe.

---

## New findings (from reviewing `ca9b311` itself)

Ranked by severity. No HIGH, no MEDIUM. Three LOW.

### 1. LOW — P19 (parallel hand-copy) — `REQUIRED` is a second, hand-maintained mirror of what the arms read

- Location: `src/bowen/ensemble/criteria.py:69-88` (the `REQUIRED` dict), consumed by `load_settings`
  (`:101-116`).
- Risk: the strict parse binds `config/bowen/criteria.md` ↔ `REQUIRED`, and
  `test_criteria_settings_are_parsed_strictly` (`tests/bowen/test_ensemble_record.py:118-133`) proves that
  binding. Nothing binds `REQUIRED` ↔ the arms' actual `settings[...]` reads. Drift in either direction is
  invisible to the default suite, because the arms run only under the deselected `ensemble` marker
  (`tests/bowen/test_phase_c_gate.py:38`; `tests/bowen/test_ensemble.py` exercises `run_criterion` only with
  synthetic criteria, never the real arms): a key the arm reads that is missing from `REQUIRED` cannot be
  put in config (strict parse rejects it) and KeyErrors only at the next ensemble run, not in the default
  suite; a key the arm no longer reads but `REQUIRED` still lists becomes silent dead config that no test
  catches. This is the same producer/consumer drift class the project has been fixing (first-gate findings
  2 and 3), one layer deeper in the fix's own new surface.
- Fix: derive `REQUIRED` from the arms (each arm declares its settings), or add a static test that walks
  `criteria.py` for `settings["…"]` / `SETTINGS["…"]` reads and asserts the set equals `REQUIRED`'s image.
- Confidence: high on the drift class; no wrong result in the tree today (records are current, suite green).

### 2. LOW — P19 (source-text corollary) — the module docstring duplicates the spell's values in prose

- Location: `src/bowen/ensemble/criteria.py:11-12` ("…declared **here** before any criterion ran") and
  `:14-15` ("a `JOB_LOSS` stressor of intensity 120 to each parent every 4 weeks from week 4"), now that the
  spell's intensity/period/start live in `config/bowen/criteria.md` (rows `spell/intensity=120.0`,
  `spell/every=4`, `spell/from=4`).
- Risk: the docstring re-states the spell's numbers as prose. Editing `config/bowen/criteria.md` (which the
  hash-staleness tests will then flag and force a regeneration of) leaves the docstring's "120 … every 4
  weeks from week 4" stale with no test tying prose to source — the same prose-vs-source drift the first
  gate's finding 2 fixed for `not_built`. "declared here" is also now misdirecting: the numbers are declared
  in config, not in the module.
- Fix: drop the literal numbers and the word "here", and point the prose at `config/bowen/criteria.md`.
- Confidence: high.

### 3. LOW (informational) — the sweep-tool import loads a second copy of the mutation tool

- Location: `tests/bowen/test_ensemble_record.py:47-55`. `_sweep` is registered under the spec name
  `"sweep_record_tool"`, but `tools/sweep_record.py:30` (`from tools.mutation_record import …`) imports the
  *real* module name `tools.mutation_record`, which is not what `sys.modules["mutation_record_tool"]` holds.
  Exec'ing `_sweep` therefore re-executes `mutation_record.py` as a distinct module, so the test now carries
  two copies of `MUTANTS`/`HASHED_TOOLS`/`Mutant`.
- Risk: harmless today — `code_hash` is path+bytes deterministic, so the sweep hash is unaffected, and the
  mutation tool has no import-time side effect. It is a fragile double-load that would run any future
  import-time side effect twice, and it muddies which copy a reader is looking at.
- Fix (optional): register the tools under their real module names (`sys.modules["tools.mutation_record"] =
  _mutants`) so `sweep_record.py`'s import resolves to the already-loaded copy, or accept the duplicate with
  a one-line comment. Not required for correctness.
- Confidence: high.

---

## Explicitly out of scope, unchanged from the prior gates

The Phase C acceptance gate still does not pass (5 criteria fail, 1 undetermined, 3 not built; several pass
only in some cells or near the frozen constants). That is a scientific/domain result reported in
`docs/phase_c_completion_report.md`, not a failure-pattern defect, and is outside this sweep's scope.

---

## Verdict

VERDICT: APPROVED

Findings A, B and 4 are correctly and completely fixed, each with a regression test and, for finding 4, a
differential verification (byte-identical records apart from the hash line, plus an exec-vs-import probe
showing every moved setting equals its former literal). The default suite is green (483 passed, 19
deselected), and nothing in the previously-approved 20 commits changed. The three new findings are all
LOW-severity guard gaps or informational drift surfaces of the P19 class the project already tracks — none
is a live defect and none produces a wrong result in the tree today. They are recorded as backlog items for
the same pre-Phase-D pass that already carries the now-closed findings.
