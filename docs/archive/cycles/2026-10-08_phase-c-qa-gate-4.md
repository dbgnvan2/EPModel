# Phase C QA gate (re-gate 3) — learning-qa failure-pattern sweep

Date: 2026-10-08
Branch: plan-phase-b (ahead of origin/plan-phase-b by 1 commit)
Reviewer: learning-qa failure-pattern sweep (inline re-gate of commit `5102e95`)

This is a re-gate. The prior gate (`docs/cycles/2026-10-08_phase-c-qa-gate-3.md`, APPROVED with three
LOW findings and no high or medium) reviewed `ca9b311`. Since then exactly one commit landed —
`5102e95` — which fixes those three low findings. This pass verifies the three fixes, reviews `5102e95`
in full, and confirms whether anything in the range regressed.

RANGE: origin/plan-phase-b..HEAD
COMMITS: 1 (`5102e95 bowen: fix the third Phase C gate's three low findings`). `origin/plan-phase-b` is
unchanged at `347b688`; the previously-approved commits are untouched.
APPLICABLE: P4 (hardcoded constants), P6 (a derived/status count not reconciled to ground truth),
P19 (producer/consumer drift, incl. its source-text corollary and the parallel-hand-copy pitfall),
P26 (the fix commit is the least-reviewed code and introduces defects), P27 (a test that cannot fail) —
plus the CLAUDE.md rules (constants-in-config, mutation-proved coverage, "a red suite is never normal").
CHECKED: the full `5102e95` diff; the three fixes read in context against their producers and consumers
(`src/bowen/ensemble/criteria.py`, `tests/bowen/test_ensemble_record.py`,
`tests/test_spec_consistency.py`, `config/bowen/criteria.md`, `tools/ensemble_record.py`,
`tools/mutation_record.py`, `tools/sweep_record.py`); the new REQUIRED-binding test traced key-by-key and
mutation-proved in both directions; the single-load test mutation-proved; a grep for literal spell numbers
in the docstring; the byte-identity of the three regenerated records against their parents; the
current-ness of the committed hashes (via the default suite's staleness tests); and the default suite.
NOT COVERED: the 19 `ensemble`-marked criterion runs (deselected by design; their committed records are
checked by the default suite), and plan §9's pending human reviews — as before.

---

## Test result

`python3 -m pytest tests/ -q` — **1 failed, 484 passed, 19 deselected** (25.96 s). **RED.**

The one failure is the regression below. The 19 `ensemble`-marked criterion tests are deselected by
`pytest.ini` by design.

---

## The regression (from reviewing `5102e95` itself)

### 1. HIGH — P6 / P26 — the suite count was bumped in two docs but not in the third; the suite is red

- Location: `CHANGELOG.md:48` — the second-gate entry ends "…apart from the hash they are byte-identical.
  **483 tests pass.**" The commit bumped the count to 485 in `CLAUDE.md:79` and
  `docs/phase_c_completion_report.md:15` and added a *new* sentence ending "485 tests pass" at
  `CHANGELOG.md:51`, but left the earlier entry's "483 tests pass" in place.
- `test_claimed_test_count_matches_the_suite` (`tests/test_spec_consistency.py:320-339`) scans every
  citing document (`CITING_DOCS`, `:37-45` — model_explainer, the proposal, `_STATUS.md`, CLAUDE.md,
  README, CHANGELOG, TODO) for `\d+ tests (?:pass|green)` and asserts each equals the collected total. It
  fails at `:339`: `AssertionError: documents claim a suite size that is not 485: {'CHANGELOG.md': [483]}`.
- This is a live, build-breaking regression introduced by the fix commit itself (P26). The count was
  propagated to two documents and a new sentence added, but the historical entry was missed — the exact
  drift class the guard exists to catch (its docstring: *"'37 tests green' sat stale in `_STATUS.md` and
  `CHANGELOG.md` while CLAUDE.md's own figure was correct"*). "485 tests pass" is also currently false in
  the plain-English sense: the suite is 485 *collected* with one of them failing.
- Fix: change `CHANGELOG.md:48` "483 tests pass" → "485 tests pass" (the guard-consistent form), then
  re-run the suite. Trivial, but it blocks the gate.
- Confidence: certain — reproduced, exit code 1.

---

## Verification of the three fixes

### Finding 1 (LOW, P19 parallel-hand-copy) — `REQUIRED` now bound to the arms in both directions: FIXED, mutation-proved

`src/bowen/ensemble/criteria.py:66-68` adds `INJECTED = frozenset({"twosome", "change", "levels",
"arms"})` — the keys the criteria table injects into an arm's settings for expanded cells, never config.
`tests/bowen/test_ensemble_record.py:143-183` adds `_setting_reads()` (an AST walk of `criteria.py` that
collects each top-level function's literal `settings["k"]` reads plus module-level `SETTINGS["s"]["k"]`
and `SPELL["k"]` reads) and `test_criteria_required_matches_what_the_arms_read`, which asserts:

1. **Forward** — every key an arm reads is in the settings that criterion is given (`:173-175`); a key the
   arm reads that is not in `REQUIRED` therefore cannot reach `criterion.settings` and fails here.
2. **Reverse** — every `REQUIRED` key is read somewhere (`:176-182`), either by an arm of its section, at
   module level, or (for `M11.C.38`'s `level_*`) by the `.items()` prefix (`:180`); the final assertion
   (`:183`) closes the other direction: nothing is read that `REQUIRED` does not declare.

I traced the mapping key-by-key against every arm and the `CRITERIA` table (all 17 sections reconcile,
including the injected keys on `M11.C.27`/`.41` and the shared arms `arm_c1`/`arm_spell_triad`), then
**mutation-proved both directions** against the verbatim source: adding an undeclared read
(`settings["bogus_key"]` in `arm_c1`) turns the test red at `:175`; adding a dead config key (`"bogus"`
to `REQUIRED["M11.C.1"]` plus the matching config row, read by nothing) turns it red at `:182`. The guard
is not a hand-copied mirror — it derives from the module's own AST, closing the drift class the finding
named.

### Finding 2 (LOW, P19 source-text corollary) — the docstring no longer restates the spell: FIXED

`src/bowen/ensemble/criteria.py:9-13` now says the readout definitions live in the module and "every
number an arm uses — horizons, scripted-act weeks, levels, starting states and the declared spell
(`M11.3`) — in ``config/bowen/criteria.md``", with the old "declared **here**", "intensity 120", "every
4 weeks from week 4" prose removed. `grep` for `120`, `every 4`, `from week 4`, and `declared here`
returns nothing in the module. No prose-vs-source drift surface remains.

### Finding 3 (LOW, informational) — the mutation tool loads once: FIXED, mutation-proved

`tests/bowen/test_ensemble_record.py:54-55` registers the loaded copy under its real import name before
the sweep tool is exec'd (`sys.modules["tools.mutation_record"] = _mutants`), so
`tools/sweep_record.py:30`'s `from tools.mutation_record import …` resolves to the already-loaded module
rather than re-executing it. `test_sweep_tool_shares_the_loaded_mutation_tool` (`:138-140`) asserts
`_sweep.Mutant is _mutants.Mutant and _sweep.run_mutant is _mutants.run_mutant`. **Mutation-proved**:
removing the registration line turns the test red (the two `Mutant` classes differ). `grep` confirms
`sweep_record.py` is the only importer of `tools.mutation_record`, so the single-load is complete.

---

## New finding (from reviewing `5102e95` itself)

Ranked by severity. One HIGH (above); one LOW below.

### 2. LOW (informational) — the fix-3 alias leaves a module registered under a name that is not its own

- Location: `tests/bowen/test_ensemble_record.py:55`. `_mutants` is loaded with
  `spec_from_file_location("mutation_record_tool", …)`, so its `__name__` and `__spec__.name` are
  `"mutation_record_tool"`; the new line registers that same object in `sys.modules` under
  `"tools.mutation_record"`. Any *future* `import tools.mutation_record` in the test process now yields a
  module whose `__name__` is `mutation_record_tool`, and a pickled `Mutant` records
  `__module__ == "mutation_record_tool"`.
- Risk: harmless today — the sweep tool is the only importer, the mutation tool has no import-time side
  effect, and `code_hash` is path+bytes deterministic. It is a mild global-state side effect the prior
  (two-copy) state did not have, and would confuse an `importlib.reload` or name-based lookup on that
  module.
- Fix (optional): register the loaded copy under its real name and set `_mutants.__name__ =
  "tools.mutation_record"` (and `__spec__.name`) so the alias is honest, or accept it with the existing
  comment. Not required for correctness.
- Confidence: high on the mechanism; no wrong result in the tree today.

---

## The regenerated records

The commit regenerates all three records. Verified two ways:

1. **Byte-identity.** `git show origin/plan-phase-b:<record>` minus the `code_hash:` line is byte-identical
   to the committed record for ensemble, mutation and sweep. Every verdict, seed count, difference, flag
   and mutant/sweep result is unchanged; only the hash line differs (the hash changes because the
   `criteria.py` docstring/`INJECTED`/comment and the regenerating run are themselves hashed inputs, while
   the run outcomes are identical).
2. **Current.** The default suite's staleness tests (`test_m134_ensemble_record_is_current`,
   `test_m111d_mutation_record_is_current`, `test_d9_sweep_record_is_current`) all pass, so the committed
   hashes match what the tools compute today.

---

## Explicitly out of scope, unchanged from the prior gates

The Phase C acceptance gate still does not pass (5 criteria fail, 1 undetermined, 3 not built; several pass
only in some cells or near the frozen constants). That is a scientific/domain result reported in
`docs/phase_c_completion_report.md`, not a failure-pattern defect, and is outside this sweep's scope.

---

## Verdict

VERDICT: REJECTED

All three low findings from gate 3 are correctly and completely fixed, each with a regression test that is
mutation-proved (both directions of the REQUIRED binding, and the single-load assertion), and the
byte-identity of the regenerated records is verified. But the fix commit itself introduced a release-blocking
regression: the suite count was propagated to `CLAUDE.md` and the completion report but not to the
historical "483 tests pass" entry in `CHANGELOG.md:48`, and `test_claimed_test_count_matches_the_suite`
now fails — the suite is red (1 failed, 484 passed, 19 deselected), against the project's standing rule that
a red suite is never normal. The fix is a one-number change in `CHANGELOG.md:48`; re-sweep the fix commit as
its own range before re-gating.
