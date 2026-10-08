# Phase C QA gate (re-gate 5) — learning-qa failure-pattern sweep

Date: 2026-10-08
Branch: plan-phase-b (ahead of origin/plan-phase-b by 1 commit)
Reviewer: learning-qa failure-pattern sweep (inline re-gate of commit `254db71`)

This is a re-gate. The prior gate (`docs/cycles/2026-10-08_phase-c-qa-gate-5.md`, APPROVED with one
LOW informational finding) reviewed `b8fa8bb`. Since then exactly one commit landed — `254db71` — which
claims to fix that one LOW finding: `tests/bowen/test_ensemble_record.py` loaded `tools/sweep_record.py`
under the alias `sweep_record_tool`, the symmetric case to the mutation-tool alias the fourth gate fixed.
This pass verifies the fix, reviews `254db71` in full, and confirms nothing else in the range regressed.

RANGE: origin/plan-phase-b..HEAD
COMMITS: 1. `254db71` — "bowen: load the sweep tool under its real module name in the tests (fifth
gate's finding)". `origin/plan-phase-b` is at `b39be15`. The commit touches only `CHANGELOG.md`,
`TODO.md`, and `tests/bowen/test_ensemble_record.py` (7 insertions, 5 deletions).
APPLICABLE: P19 (producer/consumer drift and its source-text corollary), P26 (the fix commit is the
least-reviewed code), P27 (a test that cannot fail) — plus the CLAUDE.md rules ("a red suite is never
normal", mutation-proved coverage, the count guard).
CHECKED: the full `254db71` diff; the three touched files in context; the sweep tool and mutation tool
sources (`tools/sweep_record.py`, `tools/mutation_record.py`) to confirm the module-name change has no
behavioural effect; a whole-tree grep for the removed alias and for any consumer of the synthetic
`tools.sweep_record` name; the count-guard test; the module-name assertion mutation-proved against the
real file (revert the alias → red → restore); and the default suite with its exit code.
NOT COVERED: the 19 `ensemble`-marked criterion runs (deselected by design; their committed records are
checked by the default suite), and plan §9's pending human reviews — as before.

---

## Test result

`python3 -m pytest tests/ -q` — **485 passed, 19 deselected** (26.39 s), **exit code 0. GREEN.**

The 19 `ensemble`-marked criterion tests are deselected by `pytest.ini` by design. The exit code was
checked explicitly (no pipe to `tail`), and is 0.

---

## Verification of the fix (gate-5's one LOW finding)

### Finding 1 (LOW, informational) — the sweep tool loaded under the alias `sweep_record_tool`: FIXED, mutation-proved

`254db71` changes `tests/bowen/test_ensemble_record.py:47` from
`spec_from_file_location("sweep_record_tool", …)` to `spec_from_file_location("tools.sweep_record", …)`,
so the sweep tool's `__name__` now matches the name it is registered under — the symmetric case to the
mutation tool, already loaded as `tools.mutation_record` since the fourth gate.

- The single-load property of the mutation tool is preserved: `tools/sweep_record.py:30` does
  `from tools.mutation_record import ENSEMBLE_RECORD, Mutant, run_mutant`, which resolves to the
  already-loaded `_mutants` copy (`test_sweep_tool_shares_the_loaded_mutation_tool` still asserts
  `_sweep.Mutant is _mutants.Mutant` and passes). Still one copy, not two.
- `test_sweep_tool_shares_the_loaded_mutation_tool` (`:138-142`) now also asserts
  `_sweep.__name__ == "tools.sweep_record" and sys.modules["tools.sweep_record"] is _sweep`, pinning the
  invariant rather than leaving it as an unasserted style point.
- **Mutation-proved against the real file:** reverting `:47` to the verbatim alias `sweep_record_tool`
  turns the test red (`MUTATED-EXIT: 1`, "1 failed") at the new `:142` assertion; the file was restored
  and the test passes again (`1 passed`), and `git status --porcelain` is empty after the run.
- `grep -rn "sweep_record_tool"` returns nothing anywhere in the tree; the alias string is gone.

The module-name change is behaviourally inert: `tools/sweep_record.py` references neither `__name__` nor
`__module__`, and derives `REPO`/`RECORD`/`HASHED_TOOLS` from `__file__`, so the name only affects the
`sys.modules` key and the module's `__name__`. No consumer imports `tools.sweep_record` (grep finds
none), so the real-name registration has no downstream effect — the fix is exactly the consistency it
claims, not a behaviour change.

---

## New findings from reviewing `254db71` itself

None.

The commit's only executable change is the test-file edit, which is itself the tested guard: the new
assertion is provable-failing (mutation-proved above), and the name change is inert. The `CHANGELOG.md`
and `TODO.md` edits are prose only and introduce no `\d+ tests pass/green` claim (the count guard still
passes: `test_claimed_test_count_matches_the_suite` → 1 passed). No P26 regression, no P19 drift, no
P27 dead test.

### Count-guard reconciliation

`grep -rniE '[0-9]+ tests (pass|green)'` across the three citing docs returns exactly three hits, each
"485 tests pass" (`CHANGELOG.md:51`, `CLAUDE.md:79`, `docs/phase_c_completion_report.md:15`). The new
CHANGELOG bullet ("From the fourth and fifth gates: …") carries no number, so the guard's single
correct value of 485 is unchanged and matches the measured suite.

---

## The records

`254db71` regenerates no records, and none are needed: its file changes are `CHANGELOG.md` (not a hashed
input), `TODO.md` (not hashed), and `tests/bowen/test_ensemble_record.py` (tests are not hashed inputs).
The three staleness tests (`test_m134_ensemble_record_is_current`,
`test_m111d_mutation_record_is_current`, `test_d9_sweep_record_is_current`) pass inside the 485, so the
committed `code_hash` values still match what the tools compute today.

---

## Explicitly out of scope, unchanged from the prior gates

The Phase C acceptance gate still does not pass (5 criteria fail, 1 undetermined, 3 not built; several
pass only in some cells or near the frozen constants). That is a scientific/domain result reported in
`docs/phase_c_completion_report.md`, not a failure-pattern defect, and is outside this sweep's scope.

---

## Verdict

VERDICT: APPROVED
